using System.Collections.Generic;
using Libcuda.Versions;

namespace Libptx.ObjectModel
{
    internal class Module
    {
        public SoftwareIsa Version { get; set; }
        public HardwareIsa Target { get; set; }
        public Texmode Texmode { get; set; }
        public bool EmulateDoubles { get; set; }

        public List<Entry> Entries { get; set; }
        public List<Func> Funcs { get; set; }

        public Module()
        {
            Version = SoftwareIsa.PTX_21;
            Target = HardwareIsa.SM_10;
            Texmode = Texmode.Unified;
            EmulateDoubles = false;

            Entries = new List<Entry>();
            Funcs = new List<Func>();
        }
    }
}
