using NUnit.Framework;
using XenoGears.Functional;

namespace Libptx.Playground.Parse
{
    [TestFixture]
    public class Cubin : BaseCubinTests
    {
        [Test]
        public void matmul()
        {
            2.TimesDo(() =>
            {
                var expected = ReferenceBinary();
                var module = expected.ParseCubin();
                module.Validate();
                var actual = module.RenderCubin();
                VerifyResult(expected, actual);
            });
        }
    }
}