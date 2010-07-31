using NUnit.Framework;
using XenoGears.Playground.Framework;
using XenoGears.Functional;

namespace Libptx.Playground.Emit
{
    [TestFixture]
    public class Tests : BaseTests
    {
        [Test]
        public void matmul()
        {
            2.TimesDo(() =>
            {
                var adhoc = AdHoc.matmul();
                adhoc.Validate();
                var s_adhoc = adhoc.RenderPtx();

                var edsl = Edsl.matmul();
                edsl.Validate();
                var s_edsl = adhoc.RenderPtx();

                VerifyResult(s_adhoc, s_edsl);
            });
        }
    }
}