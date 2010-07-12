using Libcuda.DataTypes;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using XenoGears.Assertions;

namespace Libptx
{
    public class Tuning : Atom
    {
        public int Maxnreg { get; set; }
        public dim3 Maxntid { get; set; }
        public dim3 Reqntid { get; set; }
        public int Minnctapersm { get; set; }
        public int Maxnctapersm { get; set; }

        public override void Validate()
        {
            (Maxnreg.AssertThat(i => i >= 0) > 0).AssertImplies(Ctx.Version >= SoftwareIsa.PTX_13);
            (Maxntid != null).AssertImplies(Ctx.Version >= SoftwareIsa.PTX_13);
            (Reqntid != null).AssertImplies(Ctx.Version >= SoftwareIsa.PTX_21);
            (Minnctapersm.AssertThat(i => i >= 0) > 0).AssertImplies(Ctx.Version >= SoftwareIsa.PTX_20);
            (Maxnctapersm.AssertThat(i => i >= 0) > 0).AssertImplies(Ctx.Version >= SoftwareIsa.PTX_13);
        }
    }
}