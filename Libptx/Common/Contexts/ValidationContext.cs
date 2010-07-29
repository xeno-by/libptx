using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using XenoGears;
using XenoGears.Assertions;
using XenoGears.Traits.Disposable;

namespace Libptx.Common.Contexts
{
    [DebuggerNonUserCode]
    public class ValidationContext : Context
    {
        public static ValidationContext Current { get { return "libptx-vctx".TlsGetOrCreate(() => new Stack<ValidationContext>()).FirstOrDefault(); } }
        public static IDisposable Push(ValidationContext ctx)
        {
            var stk = "libptx-vctx".TlsGetOrCreate(() => new Stack<ValidationContext>());
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

        public ValidationContext(Module module)
            : base(module)
        {
        }
    }
}