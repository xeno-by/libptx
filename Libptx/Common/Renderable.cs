using System;
using System.Diagnostics;
using System.IO;
using System.Text;
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

        public static String PeekRenderPtx(this Renderable renderable)
        {
            return PeekRenderPtx(renderable, RenderPtxContext.Current);
        }

        public static String PeekRenderPtx(this Renderable renderable, Module ctx)
        {
            var curr = RenderPtxContext.Current;
            if (curr != null && curr.Module == ctx)
            {
                return PeekRenderPtx(renderable);
            }
            else
            {
                return PeekRenderPtx(renderable, new RenderPtxContext(ctx));
            }
        }

        public static String PeekRenderPtx(this Renderable renderable, RenderPtxContext ctx)
        {
            if (renderable == null) return null;
            if (ctx == null) return PeekRenderPtx(renderable);

            using (RenderPtxContext.Push(ctx))
            {
                var buf = new StringBuilder();
                using (ctx.OverrideBuf(buf))
                {
                    renderable.RenderPtx(ctx);
                    var ptx = ctx.Result;
                    return ptx;
                }
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

        public static byte[] PeekRenderCubin(this Renderable renderable)
        {
            return PeekRenderCubin(renderable, RenderCubinContext.Current);
        }

        public static byte[] PeekRenderCubin(this Renderable renderable, Module ctx)
        {
            var curr = RenderCubinContext.Current;
            if (curr != null && curr.Module == ctx)
            {
                return PeekRenderCubin(renderable);
            }
            else
            {
                return PeekRenderCubin(renderable, new RenderCubinContext(ctx));
            }
        }

        public static byte[] PeekRenderCubin(this Renderable renderable, RenderCubinContext ctx)
        {
            if (renderable == null) return null;
            if (ctx == null) return PeekRenderCubin(renderable);

            using (RenderCubinContext.Push(ctx))
            {
                var buf = new MemoryStream();
                using (ctx.OverrideBuf(buf))
                {
                    renderable.RenderCubin(ctx);
                    var cubin = ctx.Result;
                    return cubin;
                }
            }
        }
    }
}
