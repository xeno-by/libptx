using System.Diagnostics;
using System.IO;
using Libcuda.DataTypes;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using XenoGears.Assertions;

namespace Libptx.Common.Performance
{
    [DebuggerNonUserCode]
    public class Tuning : Atom
    {
        [Affix("maxnreg", SoftwareIsa.PTX_13)] public int Maxnreg { get; set; }
        [Affix("maxntid", SoftwareIsa.PTX_13)] public dim3 Maxntid { get; set; }
        [Affix("reqntid", SoftwareIsa.PTX_21)] public dim3 Reqntid { get; set; }
        [Affix("minnctapersm", SoftwareIsa.PTX_20)] public int Minnctapersm { get; set; }
        [Affix("maxnctapersm", SoftwareIsa.PTX_13)] public int Maxnctapersm { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            (Maxnreg >= 0).AssertTrue();
            (Minnctapersm >= 0).AssertTrue();
            (Maxnctapersm >= 0).AssertTrue();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            if (Maxnreg != 0) writer.WriteLine(".maxnreg {0}", Maxnreg);
            if (Maxntid != new dim3()) writer.WriteLine(".maxntid {0}, {1}, {2}", Maxntid.X, Maxntid.Y, Maxntid.Z);
            if (Reqntid != new dim3()) writer.WriteLine(".reqntid {0}, {1}, {2}", Reqntid.X, Reqntid.Y, Reqntid.Z);
            if (Minnctapersm != 0) writer.WriteLine(".minnctapersm {0}", Minnctapersm);
            if (Maxnctapersm != 0) writer.WriteLine(".maxnctapersm {0}", Maxnctapersm);
        }
    }
}