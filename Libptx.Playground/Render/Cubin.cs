using Libptx.Playground.Emit;
using NUnit.Framework;
using XenoGears.Functional;

namespace Libptx.Playground.Render
{
    [TestFixture]
    public class Cubin : BaseCubinTests
    {
        [Test]
        public void matmul()
        {
            2.TimesDo(() =>
            {
                var module = AdHoc.matmul();
                module.Validate();
                var cubin = module.RenderCubin();
                VerifyResult(cubin);
            });
        }
    }
}