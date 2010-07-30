using Libptx.Playground.Emit;
using NUnit.Framework;

namespace Libptx.Playground.Render
{
    [TestFixture]
    public class Ptx : BasePtxTests
    {
        [Test, Category("Hot")]
        public void matmul()
        {
            var module = AdHoc.matmul();
            module.Validate();
            var ptx = module.RenderPtx();
            VerifyResult(ptx);
        }
    }
}
