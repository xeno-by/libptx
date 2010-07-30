using NUnit.Framework;

namespace Libptx.Playground.Parse
{
    [TestFixture]
    public class Ptx : BasePtxTests
    {
        [Test]
        public void matmul()
        {
            var expected = ReferenceText();
            var module = expected.ParsePtx();
            module.Validate();
            var actual = module.RenderPtx();
            VerifyResult(expected, actual);
        }
    }
}
