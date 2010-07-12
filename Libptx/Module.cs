using System.Collections.Generic;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx
{
    public class Module
    {
        private SoftwareIsa _version;
        public SoftwareIsa Version
        {
            get { return _version; }
            set
            {
                (Version < SoftwareIsa.PTX_15).AssertImplies(UnifiedTexturing == false);
                _version = value;
            }
        }

        private HardwareIsa _target;
        public HardwareIsa Target
        {
            get { return _target; }
            set
            {
                (value >= HardwareIsa.SM_13).AssertImplies(EmulateDoubles == false);
                _target = value;
            }
        }

        private bool _emulateDoubles;
        public bool EmulateDoubles
        {
            get { return _emulateDoubles; }
            set
            {
                (value == true).AssertImplies(Target < HardwareIsa.SM_13);
                _emulateDoubles = value;
            }
        }

        private bool _unifiedTexturing;
        public bool UnifiedTexturing
        {
            get { return _unifiedTexturing; }
            set
            {
                (value == true).AssertImplies(Version >= SoftwareIsa.PTX_15);
                _unifiedTexturing = value;
            }
        }

        public IList<Entry> Entries { get; private set; }
        public IList<Func> Funcs { get; private set; }

        public Module()
        {
            Version = SoftwareIsa.PTX_21;
            Target = HardwareIsa.SM_10;

            EmulateDoubles = false;
            UnifiedTexturing = true;

            Entries = new List<Entry>();
            Funcs = new List<Func>();
        }
    }
}
