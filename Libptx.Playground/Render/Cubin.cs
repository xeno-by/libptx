using Libptx.Playground.Emit;
using NUnit.Framework;

namespace Libptx.Playground.Render
{
    [TestFixture]
    public class Cubin : BaseCubinTests
    {
        [Test]
        public void matmul()
        {
            var module = AdHoc.matmul();
            module.Validate();
            var cubin = module.RenderCubin();
            VerifyResult(cubin);
        }
    }
}