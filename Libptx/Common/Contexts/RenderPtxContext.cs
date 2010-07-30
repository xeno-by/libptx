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

        public StringBuilder Buf { get; private set; }
        public DelayedWriter Writer { get; private set; }

        public RenderPtxContext(Module module)
            : base(module)
        {
            var core = new StringWriter(Buf);
            Writer = core.Indented().Delayed();
        }
    }
}