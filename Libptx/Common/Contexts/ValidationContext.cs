using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using XenoGears.Assertions;
using XenoGears.Traits.Disposable;

namespace Libptx.Common.Contexts
{
    [DebuggerNonUserCode]
    public class ValidationContext : Context
    {
        [ThreadStatic]
        private static Stack<ValidationContext> _stack = new Stack<ValidationContext>();
        public static ValidationContext Current { get { return _stack.FirstOrDefault(); } }
        public static IDisposable Push(ValidationContext ctx)
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

        public ValidationContext(Module module)
            : base(module)
        {
        }
    }
}