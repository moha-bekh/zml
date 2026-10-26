const std = @import("std");
const zml = @import("zml");

pub const std_options: std.Options = .{ .log_level = .warn };

const Args = struct {
    mode: enum { auto, cpu, cuda, capabilities } = .auto,
};

fn answer(input: zml.Tensor) zml.Tensor {
    return input.addConstant(22);
}

pub fn main(init: std.process.Init) void {
    run(init) catch |err| {
        std.debug.print("error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(init: std.process.Init) !void {
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    std.debug.print("probe_started mode={s}\n", .{@tagName(args.mode)});
    if (args.mode == .capabilities) {
        std.debug.print("cuda_enabled={}\n", .{zml.Target.cuda.isEnabled()});
        return;
    }
    const platform = switch (args.mode) {
        .auto => try zml.Platform.auto(init.gpa, init.io, .{}),
        .cpu => try zml.Platform.init(init.gpa, init.io, .cpu, .{}),
        .cuda => try zml.Platform.init(init.gpa, init.io, .cuda, .{}),
        .capabilities => unreachable,
    };
    defer platform.deinit(init.gpa, init.io);
    std.debug.print("selected_backend={s}\n", .{@tagName(platform.target)});
    const input: zml.Tensor = .init(.{}, .i32);
    var exe = try platform.compileFn(init.gpa, init.io, answer, .{input}, .{});
    defer exe.deinit();
    var input_buffer = try zml.Buffer.scalar(init.io, platform, 20, .i32);
    defer input_buffer.deinit();
    var result = try exe.eval(init.gpa, init.io, .{input_buffer});
    defer result.deinit();
    if (try result.getValue(i32, init.io) != 42) return error.WrongResult;
    std.debug.print("result=42\n", .{});
}
