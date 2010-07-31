using NUnit.Framework;
using XenoGears.Functional;

namespace Libptx.Playground.Parse
{
    [TestFixture]
    public class Ptx : BasePtxTests
    {
        [Test]
        public void matmul()
        {
            2.TimesDo(() =>
            {
                var expected = ReferenceText();
                var module = expected.ParsePtx();
                module.Validate();
                var actual = module.RenderPtx();
                VerifyResult(expected, actual);
            });
        }
    }
}
