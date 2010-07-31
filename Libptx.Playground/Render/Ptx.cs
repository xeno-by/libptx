using Libptx.Playground.Emit;
using NUnit.Framework;
using XenoGears.Functional;

namespace Libptx.Playground.Render
{
    [TestFixture]
    public class Ptx : BasePtxTests
    {
        [Test, Category("Hot")]
        public void matmul()
        {
            2.TimesDo(() =>
            {
                var module = AdHoc.matmul();
                module.Validate();
                var ptx = module.RenderPtx();
                VerifyResult(ptx);
            });
        }
    }
}
