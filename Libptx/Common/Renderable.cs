using System;
using System.Diagnostics;
using Libptx.Common.Contexts;
using XenoGears.Assertions;

namespace Libptx.Common
{
    public interface Renderable
    {
        void RenderPtx();
        void RenderCubin();
    }

    [DebuggerNonUserCode]
    public static class RenderableExtensions
    {
        public static void RenderPtx(this Renderable renderable)
        {
            RenderPtx(renderable, RenderPtxContext.Current);
        }

        public static void RenderPtx(this Renderable renderable, Module ctx)
        {
            var curr = RenderPtxContext.Current;
            if (curr != null && curr.Module == ctx)
            {
                RenderPtx(renderable);
            }
            else
            {
                RenderPtx(renderable, new RenderPtxContext(ctx));
            }
        }

        public static void RenderPtx(this Renderable renderable, RenderPtxContext ctx)
        {
            if (renderable == null) return;
            if (ctx == null) RenderPtx(renderable);

            using (RenderPtxContext.Push(ctx))
            {
                renderable.RenderPtx();
            }
        }

        public static String RunRenderPtx(this Renderable renderable)
        {
            return RunRenderPtx(renderable, RenderPtxContext.Current);
        }

        public static String RunRenderPtx(this Renderable renderable, Module ctx)
        {
            var curr = RenderPtxContext.Current;
            if (curr != null && curr.Module == ctx)
            {
                return RunRenderPtx(renderable);
            }
            else
            {
                return RunRenderPtx(renderable, new RenderPtxContext(ctx));
            }
        }

        public static String RunRenderPtx(this Renderable renderable, RenderPtxContext ctx)
        {
            if (renderable == null) return null;
            if (ctx == null) return RunRenderPtx(renderable);

            var skip = ctx.Buf.Length;
            using (RenderPtxContext.Push(ctx))
            {
                renderable.RenderPtx(ctx);
                var ptx = ctx.Buf.ToString(skip, ctx.Buf.Length - skip);
                return ptx;
            }
        }

        public static void RenderCubin(this Renderable renderable)
        {
            RenderCubin(renderable, RenderCubinContext.Current);
        }

        public static void RenderCubin(this Renderable renderable, Module ctx)
        {
            var curr = RenderCubinContext.Current;
            if (curr != null && curr.Module == ctx)
            {
                RenderCubin(renderable);
            }
            else
            {
                RenderCubin(renderable, new RenderCubinContext(ctx));
            }
        }

        public static void RenderCubin(this Renderable renderable, RenderCubinContext ctx)
        {
            if (renderable == null) return;
            if (ctx == null) RenderCubin(renderable);

            using (RenderCubinContext.Push(ctx))
            {
                renderable.RenderCubin();
            }
        }

        public static byte[] RunRenderCubin(this Renderable renderable)
        {
            return RunRenderCubin(renderable, RenderCubinContext.Current);
        }

        public static byte[] RunRenderCubin(this Renderable renderable, Module ctx)
        {
            var curr = RenderCubinContext.Current;
            if (curr != null && curr.Module == ctx)
            {
                return RunRenderCubin(renderable);
            }
            else
            {
                return RunRenderCubin(renderable, new RenderCubinContext(ctx));
            }
        }

        public static byte[] RunRenderCubin(this Renderable renderable, RenderCubinContext ctx)
        {
            if (renderable == null) return null;
            if (ctx == null) return RunRenderCubin(renderable);

            var skip = (int)ctx.Buf.Length;
            var pos = ctx.Buf.Position;
            using (RenderCubinContext.Push(ctx))
            {
                renderable.RenderCubin(ctx);

                var cnt = ctx.Buf.Position;
                var cubin = new byte[cnt - pos];
                var read = ctx.Buf.Read(cubin, skip, cubin.Length);
                (read == cubin.Length).AssertTrue();

                return cubin;
            }
        }
    }
}
