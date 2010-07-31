using Libptx.Playground.Emit;
using NUnit.Framework;

namespace Libptx.Playground.Render
{
    [TestFixture]
    public class Ptx : BasePtxTests
    {
        protected override void matmul_impl()
        {
            var module = AdHoc.matmul();
            module.Validate();
            var ptx = module.RenderPtx();
            VerifyResult(ptx);
        }
    }
}
