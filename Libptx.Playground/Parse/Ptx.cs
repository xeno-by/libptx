using NUnit.Framework;

namespace Libptx.Playground.Parse
{
    [TestFixture]
    public class Ptx : BasePtxTests
    {
        protected override void matmul_impl()
        {
            var expected = ReferenceText();
            var module = expected.ParsePtx();
            module.Validate();
            var actual = module.RenderPtx();
            VerifyResult(expected, actual);
        }
    }
}
