using System.Collections.Generic;
using Libcuda.Versions;

namespace Libptx
{
    public class Module
    {
        public SoftwareIsa Version { get; set; }
        public HardwareIsa Target { get; set; }

        // todo. textmodes => swisa >= PTX_15
        public bool EmulateDoubles { get; set; }
        public bool UnifiedTexturing { get; set; }

        public List<Entry> Entries { get; set; }
        public List<Func> Funcs { get; set; }

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
