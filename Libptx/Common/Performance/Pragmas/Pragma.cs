using System;
using System.IO;
using Libptx.Reflection;
using XenoGears.Assertions;

namespace Libptx.Common.Performance.Pragmas
{
    public abstract class Pragma : Atom
    {
        public String Signature
        {
            get { return this.Signature().AssertNotNull(); }
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            writer.WriteLine(".pragma \"{0}\";", Signature);
        }
    }
}