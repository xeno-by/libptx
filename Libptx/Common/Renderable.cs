using System.IO;

namespace Libptx.Common
{
    public interface Renderable
    {
        void RenderAsPtx(TextWriter writer);
    }
}