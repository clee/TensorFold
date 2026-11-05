//! Qwen3.5 PP: one stream's cache per rank, every round carried once around a fabric stage ring.
const std = @import("std");
const cuda = @import("cuda");
const stage = @import("stage");
const E = @import("cuda_engine.zig");

/// The rows one forward may carry (a prompt chunk), E.max_rows.
pub const max_rows = E.max_rows;
/// The most stages a ring takes: --pp allows 1 to 8.
pub const most_ranks = 8;

/// A frame's fixed payload: the op, an open's capacity, the cache position and the reply flag.
const meta_bytes = 24;

const Op = enum(u32) { open = 1, close = 2, forward = 3 };

const Meta = struct {
    op: Op,
    cap: u32 = 0,
    pos0: u64 = 0,
    /// forward: this pass ends a call and the last stage answers with its logits.
    final: bool = false,
};

fn encodeMeta(m: Meta) [meta_bytes]u8 {
    var out: [meta_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, out[0..4], @backingInt(m.op), .little);
    std.mem.writeInt(u32, out[4..8], m.cap, .little);
    std.mem.writeInt(u64, out[8..16], m.pos0, .little);
    std.mem.writeInt(u32, out[16..20], @intFromBool(m.final), .little);
    return out;
}

fn decodeMeta(raw: []const u8) !Meta {
    if (raw.len < meta_bytes) return error.PipelineStageFrame;
    const op = std.enums.fromInt(Op, std.mem.readInt(u32, raw[0..4], .little)) orelse return error.PipelineStageFrame;
    const final = std.mem.readInt(u32, raw[16..20], .little);
    if (final > 1) return error.PipelineStageFrame;
    return .{ .op = op, .cap = std.mem.readInt(u32, raw[4..8], .little), .pos0 = std.mem.readInt(u64, raw[8..16], .little), .final = final != 0 };
}

/// The engine's --pp setup, or null on one card. Refusals happen before the ring opens.
pub fn openEngine(gpa: std.mem.Allocator, io: std.Io, parts: usize, rank: usize, e: *E.Engine, drafts: bool, vision: bool, slide: bool, pages: bool) !?*Pp {
    _ = io;
    if (parts <= 1) return null;
    if (rank >= parts) return error.BadRank;
    if (drafts) return error.PipelineStageDraftsUnsupported;
    if (vision) return error.PipelineStageVisionUnsupported;
    if (slide) return error.PipelineStageLearningUnsupported;
    if (pages) return error.PipelineStagePagesUnsupported;
    var s: stage.config.Settings = undefined;
    stage.config.parse(stage.config.Env{}, &s) catch |err| switch (err) {
        error.MissingStageSetting => return error.PipelineStageNoRing,
        else => return err,
    };
    if (s.world != parts) return error.PipelineStageWorldMismatch;
    if (s.rank != rank) return error.PipelineStageRankMismatch;
    return try Pp.open(gpa, s.options(fingerprint(e.g)), e);
}

/// One plain decode window through the ring, sampled from the last stage's row.
pub fn verifyOne(p: *Pp, e: *E.Engine, cache: *E.Cache, gpa: std.mem.Allocator, w: anytype, o: anytype) !void {
    const draws = @import("draw.zig");
    if (w.rows() != 1 or w.held != 0 or w.parents != null) return error.PipelineStageDraftsUnsupported;
    if (w.positions[0] != cache.len + 1) return error.InvalidQwenPosition;
    try p.forward(e, &.{w.pending}, null, true);
    const mask = if (w.masks.len > 0) w.masks[0..draws.mask_words] else &.{};
    o.sampled[0] = try draws.draw(gpa, cache.logits, w.stream.sampling, w.positions[0], mask);
    if (w.tokens.len > 0) @memcpy(o.drafts[0..w.tokens.len], w.tokens);
}

/// The checkpoint geometry every rank of a ring must agree on before any frame.
pub fn fingerprint(g: E.Geometry) [32]u8 {
    var out: [32]u8 = undefined;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("TF qwen3.5 stage ring v1");
    var ints: [8]u64 = .{ g.hidden, g.intermediate, g.layers, g.vocab, g.k_heads, g.v_heads, g.q_heads, g.kv_heads };
    h.update(std.mem.sliceAsBytes(&ints));
    h.final(&out);
    return out;
}

pub const Pp = struct {
    gpa: std.mem.Allocator,
    rank: u32,
    parts: usize,
    stage: E.Stage,
    ring: stage.Ring,
    hidden: usize,
    vocab: usize,
    /// A round's frame: its metadata, the token ids, then hidden rows or the reply's logits.
    buf: []u8,
    /// A follower's decoding of the frame's ids, handed to the engine's forward.
    ids: []u32,
    round: u64 = 0,
    cache: ?*E.Cache = null,

    pub fn open(gpa: std.mem.Allocator, o: stage.Options, e: *E.Engine) !*Pp {
        if (o.world < 2 or o.world > most_ranks or o.rank >= o.world) return error.BadRank;
        const p = try gpa.create(Pp);
        errdefer gpa.destroy(p);
        const rows = meta_bytes + max_rows * 4;
        const hidden = 2 * max_rows * e.g.hidden * 2;
        const buf = try gpa.alloc(u8, rows + @max(hidden, e.g.vocab * 2));
        errdefer gpa.free(buf);
        const ids = try gpa.alloc(u32, max_rows);
        errdefer gpa.free(ids);
        p.* = .{ .gpa = gpa, .rank = o.rank, .parts = o.world, .stage = E.Stage.of(e.g.layers, o.world, @intCast(o.rank)), .ring = undefined, .hidden = e.g.hidden, .vocab = e.g.vocab, .buf = buf, .ids = ids };
        p.ring = try stage.Ring.open(o);
        return p;
    }

    pub fn deinit(p: *Pp) void {
        if (p.cache) |c| {
            c.deinit(p.gpa);
            p.gpa.destroy(c);
        }
        p.cache = null;
        if (p.rank == 0) p.ring.stop();
        p.ring.close();
        p.gpa.free(p.ids);
        p.gpa.free(p.buf);
        p.gpa.destroy(p);
    }

    pub fn openStream(p: *Pp, e: *E.Engine, cap: usize) !void {
        if (p.rank != 0) return error.PipelineStageNotServer;
        const c = try p.gpa.create(E.Cache);
        errdefer p.gpa.destroy(c);
        c.* = try E.Cache.init(p.gpa, e, cap);
        p.cache = c;
        try p.plan(.open, @intCast(cap));
    }

    pub fn closeStream(p: *Pp) !void {
        if (p.rank != 0) return error.PipelineStageNotServer;
        try p.plan(.close, 0);
        if (p.cache) |c| {
            c.deinit(p.gpa);
            p.gpa.destroy(c);
        }
        p.cache = null;
    }

    /// One pass of `ids` through every stage: the reply's logits land in rank 0's cache when `final`.
    pub fn forward(p: *Pp, e: *E.Engine, ids: []const u32, logits_host: ?[]u8, final: bool) !void {
        if (p.rank != 0) return error.PipelineStageNotServer;
        const cache = p.cache orelse return error.PipelineStageNoSequence;
        if (ids.len == 0 or ids.len > max_rows) return error.UnexpectedTensor;
        const n = ids.len;
        const r = p.round + 1;
        const at = p.begin(.{ .op = .forward, .pos0 = cache.len, .final = final }, ids);
        const part: E.Part = .{ .cache = cache, .rows = n };
        try e.forward(&.{part}, ids, .{ .stage = false }); // a ring of two or more stages never ends here
        try p.putHidden(e, n);
        try p.ring.send(.rows, r, @intCast(n), &.{p.buf[0 .. at + 2 * n * p.hidden * 2]});
        const h = try p.ring.recv(.rows, r, p.buf);
        p.round = r;
        if (!final) return;
        if (h.bytes < at) return error.PipelineStageFrame;
        const row = p.buf[at..][0 .. @as(usize, @intCast(h.bytes)) - at];
        if (row.len != p.vocab * 2) return error.PipelineStageFrame;
        try copyLogits(cache, row);
        if (logits_host) |lh| @memcpy(lh[0..row.len], row);
    }

    /// A follower's loop: open and close frames, forwards, until the server stops the ring.
    pub fn follow(p: *Pp, e: *E.Engine) !void {
        if (p.rank == 0) return error.PipelineStageNotServer;
        while (true) {
            const r = p.round + 1;
            const h = p.ring.recv(.rows, r, p.buf) catch |err| switch (err) {
                error.Stopped, error.Closed => return,
                else => return err,
            };
            p.round = r;
            const m = try decodeMeta(p.buf[0..meta_bytes]);
            switch (m.op) {
                .open => {
                    if (p.cache) |c| {
                        c.deinit(p.gpa);
                        p.gpa.destroy(c);
                    }
                    const c = try p.gpa.create(E.Cache);
                    errdefer p.gpa.destroy(c);
                    c.* = try E.Cache.init(p.gpa, e, m.cap);
                    p.cache = c;
                    try p.ring.send(.rows, r, h.rows, &.{p.buf[0..meta_bytes]});
                },
                .close => {
                    if (p.cache) |c| {
                        c.deinit(p.gpa);
                        p.gpa.destroy(c);
                    }
                    p.cache = null;
                    try p.ring.send(.rows, r, h.rows, &.{p.buf[0..meta_bytes]});
                },
                .forward => try p.relay(e, r, h.rows, m),
            }
        }
    }

    /// A follower's pass: the rows from the stage before, its own layers, then onward to the next.
    fn relay(p: *Pp, e: *E.Engine, r: u64, n: u32, m: Meta) !void {
        const cache = p.cache orelse return error.PipelineStageNoSequence;
        if (cache.len != m.pos0) return error.PipelineStagePositionDrift;
        p.beginMeta(m);
        const at = meta_bytes + @as(usize, n) * 4;
        for (0..n) |i| p.ids[i] = std.mem.readInt(u32, p.buf[meta_bytes + 4 * i ..][0..4], .little);
        try p.getHidden(e, n);
        const part: E.Part = .{ .cache = cache, .rows = @intCast(n) };
        try e.forward(&.{part}, p.ids[0..n], if (p.stage.last and m.final) E.Keep.last else .{ .stage = false });
        if (!p.stage.last) {
            try p.putHidden(e, n);
            try p.ring.send(.rows, r, n, &.{p.buf[0 .. at + 2 * @as(usize, n) * p.hidden * 2]});
        } else if (m.final) {
            const row = std.mem.sliceAsBytes(cache.logits);
            @memcpy(p.buf[at..][0..row.len], row);
            try p.ring.send(.rows, r, n, &.{p.buf[0 .. at + row.len]});
        } else try p.ring.send(.rows, r, n, &.{p.buf[0..at]});
    }

    /// The server's open or close: every rank acts, and the frame's return is the barrier.
    fn plan(p: *Pp, op: Op, cap: u32) !void {
        const r = p.round + 1;
        p.beginMeta(.{ .op = op, .cap = cap });
        try p.ring.send(.rows, r, 0, &.{p.buf[0..meta_bytes]});
        _ = try p.ring.recv(.rows, r, p.buf);
        p.round = r;
    }

    /// Writes a frame's metadata; the ids follow at `meta_bytes`.
    fn beginMeta(p: *Pp, m: Meta) void {
        @memcpy(p.buf[0..meta_bytes], &encodeMeta(m));
    }

    /// Writes the server's metadata and ids, returning where its hidden rows go.
    fn begin(p: *Pp, m: Meta, ids: []const u32) usize {
        p.beginMeta(m);
        const at = meta_bytes + ids.len * 4;
        for (ids, 0..) |id, i| std.mem.writeInt(u32, p.buf[meta_bytes + 4 * i ..][0..4], id, .little);
        return at;
    }

    fn putHidden(p: *Pp, e: *E.Engine, n: usize) !void {
        const bytes = n * p.hidden * 2;
        const at = meta_bytes + n * 4;
        const h: cuda.DeviceBuffer = .{ .d = e.d, .ptr = e.h, .len = bytes };
        const rr: cuda.DeviceBuffer = .{ .d = e.d, .ptr = e.r, .len = bytes };
        try h.download(0, p.buf[at..][0..bytes]);
        try rr.download(0, p.buf[at + bytes ..][0..bytes]);
    }

    fn getHidden(p: *Pp, e: *E.Engine, n: usize) !void {
        const bytes = n * p.hidden * 2;
        const at = meta_bytes + n * 4;
        const h: cuda.DeviceBuffer = .{ .d = e.d, .ptr = e.h, .len = bytes };
        const rr: cuda.DeviceBuffer = .{ .d = e.d, .ptr = e.r, .len = bytes };
        try h.uploadAsync(0, p.buf[at..][0..bytes], e.stream.handle);
        try rr.uploadAsync(0, p.buf[at + bytes ..][0..bytes], e.stream.handle);
        try e.stream.synchronize();
    }
};

fn copyLogits(cache: *E.Cache, bytes: []const u8) !void {
    if (bytes.len != cache.logits.len * 2) return error.UnexpectedTensor;
    @memcpy(std.mem.sliceAsBytes(cache.logits), bytes);
}

test "stage metadata: an op, a capacity, a position and a reply round-trip, and a bad op is refused" {
    const t = std.testing;
    const m: Meta = .{ .op = .forward, .pos0 = 12_345, .final = true };
    try t.expectEqual(m, try decodeMeta(&encodeMeta(m)));
    const o: Meta = .{ .op = .open, .cap = 8192 };
    try t.expectEqual(o, try decodeMeta(&encodeMeta(o)));
    var raw = encodeMeta(m);
    raw[0] = 9;
    try t.expectError(error.PipelineStageFrame, decodeMeta(&raw));
    raw = encodeMeta(m);
    raw[16] = 2;
    try t.expectError(error.PipelineStageFrame, decodeMeta(&raw));
}

test "fingerprint: the model's geometry names a ring, and a different hidden width is another ring" {
    const t = std.testing;
    var g: E.Geometry = undefined;
    g = .{ .hidden = 2048, .intermediate = 6144, .layers = 24, .vocab = 248_320, .k_heads = 16, .v_heads = 16, .q_heads = 8, .kv_heads = 2, .eps = 1e-6, .tied = true, .context = 4096, .linear = @splat(false) };
    const a = fingerprint(g);
    try t.expectEqual(a, fingerprint(g));
    g.hidden = 2049;
    try t.expect(!std.mem.eql(u8, &a, &fingerprint(g)));
}
