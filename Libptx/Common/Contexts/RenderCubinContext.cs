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

        public RenderCubinContext(Module module)
            : base(module)
        {
            Buf = new MemoryStream();
            Writer = new BinaryWriter(Buf);
        }

        private MemoryStream Buf { get; set; }
        public byte[] Result { get { return Buf.ToArray(); } }
        public BinaryWriter Writer { get; private set; }

        public IDisposable OverrideBuf(MemoryStream new_buf)
        {
            var old_buf = Buf;
            var old_writer = Writer;

            Buf = new_buf;
            Writer = new BinaryWriter(Buf);

            return new DisposableAction(() =>
            {
                Buf = old_buf;
                Writer = old_writer;
            });
        }
    }
}