using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using XenoGears;
using XenoGears.Strings.Writers;
using XenoGears.Traits.Disposable;
using XenoGears.Assertions;

namespace Libptx.Common.Contexts
{
    [DebuggerNonUserCode]
    public class RenderPtxContext : Context
    {
        public static RenderPtxContext Current { get { return "libptx-rctx".TlsGetOrCreate(() => new Stack<RenderPtxContext>()).FirstOrDefault(); } }
        public static IDisposable Push(RenderPtxContext ctx)
        {
            var stk = "libptx-rctx".TlsGetOrCreate(() => new Stack<RenderPtxContext>());
            if (Current == ctx)
            {
                return new DisposableAction(() => {});
            }
            else
            {
                stk.Push(ctx);

                return new DisposableAction(() =>
                {
                    (Current == ctx).AssertTrue();
                    stk.Pop();
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