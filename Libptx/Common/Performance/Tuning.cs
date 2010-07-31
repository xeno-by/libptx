using System;
using System.Diagnostics;
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

        public Tuning()
        {
            Maxnreg = 0;
            Maxntid = new dim3(0, 0, 0);
            Reqntid = new dim3(0, 0, 0);
            Minnctapersm = 0;
            Maxnctapersm = 0;
        }

        public bool IsTrivial
        {
            get
            {
                var os_trivial = true;
                os_trivial &= Maxnreg == 0;
                os_trivial &= Maxntid == new dim3(0, 0, 0);
                os_trivial &= Reqntid == new dim3(0, 0, 0);
                os_trivial &= Minnctapersm == 0;
                os_trivial &= Maxnctapersm == 0;
                return os_trivial;
            }
        }

        public bool IsNotTrivial
        {
            get { return !IsTrivial; }
        }

        protected override void CustomValidate()
        {
            (Maxnreg >= 0).AssertTrue();
            (Maxntid == new dim3(0, 0, 0) || Maxntid >= new dim3(1, 1, 1)).AssertTrue();
            (Reqntid == new dim3(0, 0, 0) || Reqntid >= new dim3(1, 1, 1)).AssertTrue();
            (Minnctapersm >= 0).AssertTrue();
            (Maxnctapersm >= 0).AssertTrue();

            (Maxntid != new dim3(0, 0, 0) && Reqntid != new dim3(0, 0, 0)).AssertFalse();
            (Minnctapersm != 0 && Maxnctapersm != 0).AssertImplies(Minnctapersm <= Maxnctapersm);
        }

        protected override void RenderPtx()
        {
            if (Maxnreg != 0) writer.WriteLine(".maxnreg {0}", Maxnreg);
            if (Maxntid != new dim3()) writer.WriteLine(".maxntid {0}, {1}, {2}", Maxntid.X, Maxntid.Y, Maxntid.Z);
            if (Reqntid != new dim3()) writer.WriteLine(".reqntid {0}, {1}, {2}", Reqntid.X, Reqntid.Y, Reqntid.Z);
            if (Minnctapersm != 0) writer.WriteLine(".minnctapersm {0}", Minnctapersm);
            if (Maxnctapersm != 0) writer.WriteLine(".maxnctapersm {0}", Maxnctapersm);
        }

        protected override void RenderCubin()
        {
            throw new NotImplementedException();
        }
    }
}