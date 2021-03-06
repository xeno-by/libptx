using System;
using System.Diagnostics;
using Libptx.Reflection;
using XenoGears.Assertions;

namespace Libptx.Common.Performance.Pragmas
{
    [DebuggerNonUserCode]
    public abstract class Pragma : Atom
    {
        public String Signature
        {
            get { return this.Signature().AssertNotNull(); }
        }

        protected override void RenderPtx()
        {
            writer.WriteLine(".pragma \"{0}\";", Signature);
        }

        protected override void RenderCubin()
        {
            throw new NotImplementedException();
        }
    }
}