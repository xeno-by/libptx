using System;
using System.IO;
using Libptx.Statements;
using Libptx.Common.Annotations;
using XenoGears.Assertions;

namespace Libptx.Common.Performance.Pragmas
{
    public abstract class Pragma : Atom, Statement
    {
        public String Signature
        {
            get { return this.Signature().AssertNotNull(); }
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}