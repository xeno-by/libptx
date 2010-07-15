using NUnit.Framework;
using XenoGears.Playground.Framework;

namespace Libptx.Playground.Emit
{
    [TestFixture]
    public class Tests : BaseTests
    {
        [Test]
        public void MatMul()
        {
            // add various overloads for this ctor (same as for AtomAttribute)
            var module = new Module(SoftwareIsa.PTX_14, HardwareIsa.SM_13);

            // when vars are created, align gets defaulted to sizeof(type), space will get defaulted to param when added to Params collection
            // add appropriate ctors for Func and Entry, introduce mirrored AddFunc methods
            // also add the Signature method that provides bulk-set of the signature outside of the ctor:
            // 1) native lambda form: (uint a_width, uint a_height, uint a_raw, uint b_width, uint b_height, uint b_raw, uint c_width, uint c_height, uint c_raw) => {}
            // 2) esoteric lambda form: a => b32[12], b => b32[12], c => b32[12]
            // what about return values?
            // 1) multiple return values needs to be specified manually
            // 2) native lambda form might specify them by providing non-empty body, e.g. "(int foo) => default(float)"
            // 3) esoteric lambda form might specify them by appending "_ret" to parameter names
            var ptx = new Entry("MatMulKernel", a => b32[12], b => b32[12], c => b32[12]);
        }
    }
}
