using System;
using System.Collections.Generic;
using System.Diagnostics;
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
            Delayed = Buf.Delayed();
            Writer = Delayed.Indented();
        }

        private StringBuilder Buf { get; set; }
        private DelayedWriter Delayed { get; set; }
        public IndentedWriter Writer { get; private set; }
        public String Result { get { Delayed.IsDelayed.AssertFalse(); return Buf.ToString(); } }

        public void DelayRender(Action action)
        {
            Delayed.Delay(action);
        }

        public void CommitRender()
        {
            Delayed.Commit();
        }

        public IDisposable OverrideBuf(StringBuilder new_buf)
        {
            var old_buf = Buf;
            var old_delayed = Delayed;
            var old_writer = Writer;

            Buf = new_buf;
            Delayed = Buf.Delayed();
            Writer = Delayed.Indented();

            return new DisposableAction(() =>
            {
                Buf = old_buf;
                Delayed = old_delayed;
                Writer = old_writer;
            });
        }
    }
}