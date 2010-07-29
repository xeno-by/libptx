using System;
using System.Diagnostics;
using Libptx.Common.Contexts;

namespace Libptx.Common
{
    public interface Renderable
    {
        void RenderPtx();
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
    }
}
