//! Qwen3.5-2B across pipeline stages: the same prompt and decode row one card and n ranks, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const check = @import("check.zig");

const E = qwen.engine;

fn idsOf(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice([]const u32, gpa, text, .{});
    defer parsed.deinit();
    if (parsed.value.len == 0 or parsed.value.len > 2048) return error.BadFixture;
    return gpa.dupe(u32, parsed.value);
}

fn digest(row: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(row, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}

/// A loopback port the ring's peers can name; the probe closes before the rank binds it.
fn freePort() !u16 {
    const fd = try qwen.stage_link.net.listen("127.0.0.1", 0);
    defer qwen.stage_link.net.close(fd);
    return qwen.stage_link.net.port(fd);
}

const Worker = struct {
    driver: *const cuda.Driver,
    gpa: std.mem.Allocator,
    io: std.Io,
    model: []const u8,
    parts: usize,
    rank: usize,
    opts: qwen.stage_link.Options,
    err: ?anyerror = null,

    fn run(w: *Worker) void {
        w.main() catch |e| {
            std.debug.print("FAIL qwen35-pp: rank {d} stopped: {t}\n", .{ w.rank, e });
            w.err = e;
        };
    }

    fn main(w: *Worker) !void {
        var ctx = try cuda.Context.init(w.driver, @intCast(w.rank));
        defer ctx.deinit();
        const g = try E.Geometry.read(w.gpa, w.io, w.model);
        var e = try E.Engine.initAt(w.gpa, w.io, &ctx, w.model, false, null, E.Stage.of(g.layers, w.parts, w.rank));
        defer e.deinit();
        var o = w.opts;
        o.rank = @intCast(w.rank);
        const p = try qwen.pp.Pp.open(w.gpa, o, e);
        defer p.deinit();
        try p.follow(e);
    }
};

/// `MODEL IDS.json [PARTS]`: the prompt and one decode row through every stage, compared with one card.
pub fn run(gpu: check.Gpu, model: []const u8, ids_file: []const u8, parts: usize) !void {
    if (parts < 2 or parts > 8) return error.BadRank;
    const ids = try idsOf(gpu.gpa, gpu.io, ids_file);
    defer gpu.gpa.free(ids);
    var want_prompt: [64]u8 = undefined;
    var want_decode: [64]u8 = undefined;
    {
        var e = try E.Engine.init(gpu.gpa, gpu.io, gpu.ctx, model, false, null);
        defer e.deinit();
        var cache = try E.Cache.init(gpu.gpa, e, ids.len + 2);
        defer cache.deinit(gpu.gpa);
        try e.forward(&.{.{ .cache = &cache, .rows = ids.len }}, ids, .last);
        want_prompt = digest(std.mem.sliceAsBytes(cache.logits));
        try e.forward(&.{.{ .cache = &cache, .rows = 1 }}, ids[0..1], .last);
        want_decode = digest(std.mem.sliceAsBytes(cache.logits));
    }

    const g = try E.Geometry.read(gpu.gpa, gpu.io, model);
    var peers: [8]qwen.stage_link.Peer = undefined;
    for (0..parts) |r| peers[r] = .{ .host = "127.0.0.1", .port = try freePort() };
    const opts: qwen.stage_link.Options = .{ .rank = 0, .world = @intCast(parts), .peers = peers[0..parts], .fingerprint = qwen.pp.fingerprint(g), .key = try qwen.stage_link.secret.fromBytes("qwen35 pipeline parallel test ring"), .timeout_ns = 120 * std.time.ns_per_s };

    var workers: [8]Worker = undefined;
    var threads: [8]std.Thread = undefined;
    for (1..parts) |r| {
        workers[r] = .{ .driver = gpu.d, .gpa = gpu.gpa, .io = gpu.io, .model = model, .parts = parts, .rank = r, .opts = opts };
        threads[r] = try std.Thread.spawn(.{}, Worker.run, .{&workers[r]});
    }
    var got_prompt: [64]u8 = undefined;
    var got_decode: [64]u8 = undefined;
    {
        var e = try E.Engine.initAt(gpu.gpa, gpu.io, gpu.ctx, model, false, null, E.Stage.of(g.layers, parts, 0));
        defer e.deinit();
        const p = try qwen.pp.Pp.open(gpu.gpa, opts, e);
        errdefer p.deinit();
        try p.openStream(e, ids.len + 2);
        try p.forward(e, ids, null, true);
        got_prompt = digest(std.mem.sliceAsBytes(p.cache.?.logits));
        try p.forward(e, ids[0..1], null, true);
        got_decode = digest(std.mem.sliceAsBytes(p.cache.?.logits));
        try p.closeStream();
        p.deinit();
    }
    for (threads[1..parts]) |t| t.join();
    for (workers[1..parts]) |w| if (w.err) |e| return e;
    try check.expect(std.mem.eql(u8, &want_prompt, &got_prompt), "qwen35-pp: {d} prompt tokens, one card {s} vs {d} ranks {s}", .{ ids.len, &want_prompt, parts, &got_prompt });
    try check.expect(std.mem.eql(u8, &want_decode, &got_decode), "qwen35-pp: one decode row, one card {s} vs {d} ranks {s}", .{ &want_decode, parts, &got_decode });
    check.pass("qwen35-pp: {d} prompt tokens and one decode row, {d} ranks match one card", .{ ids.len, parts });
}
