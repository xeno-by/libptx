using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using XenoGears.Strings.Writers;
using XenoGears.Traits.Disposable;
using XenoGears.Assertions;

namespace Libptx.Common.Contexts
{
    [DebuggerNonUserCode]
    public class RenderPtxContext : Context
    {
        [ThreadStatic] private static Stack<RenderPtxContext> _stack = new Stack<RenderPtxContext>();
        public static RenderPtxContext Current { get { return _stack.FirstOrDefault(); } }
        public static IDisposable Push(RenderPtxContext ctx)
        {
            if (Current == ctx)
            {
                return new DisposableAction(() => {});
            }
            else
            {
                _stack.Push(ctx);

                return new DisposableAction(() =>
                {
                    (Current == ctx).AssertTrue();
                    _stack.Pop();
                });
            }
        }

        public RenderPtxContext(Module module)
            : base(module)
        {
            Buf = new StringBuilder();
            var core = new StringWriter(Buf);
            Writer = core.Indented().Delayed();
        }

        private StringBuilder Buf { get; set; }
        public String Result { get { Writer.IsDelayed.AssertFalse(); return Buf.ToString(); } }
        public DelayedWriter Writer { get; private set; }

        public IDisposable OverrideBuf(StringBuilder new_buf)
        {
            var old_buf = Buf;
            var old_writer = Writer;

            Buf = new_buf;
            var new_writer = new StringWriter(Buf);
            Writer = new_writer.Indented().Delayed();

            return new DisposableAction(() =>
            {
                Buf = old_buf;
                Writer = old_writer;
            });
        }
    }
}