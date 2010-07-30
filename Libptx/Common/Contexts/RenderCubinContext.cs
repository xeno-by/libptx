using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using XenoGears.Assertions;
using XenoGears.Traits.Disposable;

namespace Libptx.Common.Contexts
{
    [DebuggerNonUserCode]
    public class RenderCubinContext : Context
    {
        [ThreadStatic] private static Stack<RenderCubinContext> _stack = new Stack<RenderCubinContext>();
        public static RenderCubinContext Current { get { return _stack.FirstOrDefault(); } }
        public static IDisposable Push(RenderCubinContext ctx)
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

        public MemoryStream Buf { get; private set; }
        public BinaryWriter Writer { get; private set; }

        public RenderCubinContext(Module module)
            : base(module)
        {
            Buf = new MemoryStream();
            Writer = new BinaryWriter(Buf);
        }
    }
}