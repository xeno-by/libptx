using NUnit.Framework;

namespace Libptx.Playground.Parse
{
    [TestFixture]
    public class Cubin : BaseCubinTests
    {
        protected override void matmul_impl()
        {
            var expected = ReferenceBinary();
            var module = expected.ParseCubin();
            module.Validate();
            var actual = module.RenderCubin();
            VerifyResult(expected, actual);
        }
    }
}