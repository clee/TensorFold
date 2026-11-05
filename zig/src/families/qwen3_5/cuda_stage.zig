//! A process's contiguous slice of the Qwen3.5 decoder layers for pipeline parallelism.
const std = @import("std");

pub const Stage = struct {
    first: usize = 0,
    count: usize = 0,
    /// The last stage owns the final norm and the vocabulary head.
    last: bool = true,

    pub fn end(s: Stage) usize {
        return s.first + s.count;
    }

    /// The whole model on one process.
    pub fn whole(layers: usize) Stage {
        return .{ .first = 0, .count = layers, .last = true };
    }

    /// Contiguous stages: the first `layers % parts` ranks take one extra layer.
    pub fn of(layers: usize, parts: usize, rank: usize) Stage {
        std.debug.assert(parts >= 1 and rank < parts);
        const base = layers / parts;
        const extra = layers % parts;
        const first = rank * base + @min(rank, extra);
        const count = base + @intFromBool(rank < extra);
        return .{ .first = first, .count = count, .last = rank + 1 == parts };
    }
};

test "stages cover every layer exactly once" {
    const t = std.testing;
    for ([_]usize{ 1, 2, 3, 4, 5 }) |parts| {
        var sum: usize = 0;
        var next: usize = 0;
        for (0..parts) |r| {
            const s = Stage.of(24, parts, r);
            try t.expectEqual(next, s.first);
            try t.expectEqual(r + 1 == parts, s.last);
            sum += s.count;
            next = s.end();
        }
        try t.expectEqual(@as(usize, 24), sum);
    }
}
