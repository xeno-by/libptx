using NUnit.Framework;

namespace Libptx.Playground.Emit
{
    [TestFixture]
    public class Tests : BaseTests
    {
        protected override void matmul_impl()
        {
            var adhoc = AdHoc.matmul();
            adhoc.Validate();
            var s_adhoc = adhoc.RenderPtx();

            var edsl = Edsl.matmul();
            edsl.Validate();
            var s_edsl = adhoc.RenderPtx();

            VerifyResult(s_adhoc, s_edsl);
        }
    }
}