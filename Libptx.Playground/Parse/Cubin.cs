using NUnit.Framework;

namespace Libptx.Playground.Parse
{
    [TestFixture]
    public class Cubin : BaseCubinTests
    {
        [Test]
        public void matmul()
        {
            var expected = ReferenceBinary();
            var module = expected.ParseCubin();
            module.Validate();
            var actual = module.RenderCubin();
            VerifyResult(expected, actual);
        }
    }
}