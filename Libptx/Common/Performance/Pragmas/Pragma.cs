using System;
using System.IO;
using Libptx.Statements;

namespace Libptx.Common.Performance.Pragmas
{
    public abstract class Pragma : Atom, Statement
    {
        public String Signature
        {
            get { throw new NotImplementedException(); }
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}