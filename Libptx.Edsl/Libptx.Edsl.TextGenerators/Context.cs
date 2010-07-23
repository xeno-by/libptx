using System.Collections.Generic;
using System.Linq;
using Libcuda.Versions;
using XenoGears.Assertions;
using XenoGears.Traits.Disposable;

namespace Libptx.Edsl.TextGenerators
{
    internal class Context : Disposable
    {
        public SoftwareIsa Version { get; set; }
        public HardwareIsa Target { get; set; }

        public Context(SoftwareIsa version, HardwareIsa target)
        {
            Version = version;
            Target = target;

            _current.Push(this);
        }

        protected override void DisposeManagedResources()
        {
            (Current == this).AssertTrue();
            _current.Pop();
        }

        private static Stack<Context> _current = new Stack<Context>();
        public static Context Current { get { return _current.Any() ? _current.Peek() : null; } }
    }
}