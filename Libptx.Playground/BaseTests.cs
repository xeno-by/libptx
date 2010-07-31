using System;
using System.Collections.ObjectModel;
using System.Linq;
using NUnit.Framework;
using XenoGears.Functional;
using XenoGears.Strings;

namespace Libptx.Playground
{
    [TestFixture]
    public abstract class BaseTests : XenoGears.Playground.Framework.BaseTests
    {
        [Test, Category("Hot")]
        public void matmul()
        {
            matmul1();
            matmul2();
        }

        private void matmul1() { matmul_impl(); }
        private void matmul2() { matmul_impl(); }
        protected abstract void matmul_impl();
    }

    [TestFixture]
    public abstract class BasePtxTests : BaseTests
    {
        protected override String ReferenceNamespace()
        {
            return typeof(TestRunner).Namespace + ".Reference";
        }

        protected override ReadOnlyCollection<String> ReferenceWannabes()
        {
            return base.ReferenceWannabes().Select(wb => wb + ".ptx").ToReadOnly();
        }

        protected override String PreprocessReference(String s_reference)
        {
            if (s_reference == null) return null;
            return s_reference.SplitLines().Select(ln =>
            {
                if (ln.Trim().IsEmpty()) return ln;
                var indent = Seq.Nats.First(i => ln[i] != ' ');
                var s_indent = ln.Slice(0, indent);
                ln = ln.Slice(indent);
                while (ln.Contains("  ")) ln = ln.Replace("  ", " ");
                return s_indent + ln;
            }).StringJoin(Environment.NewLine);
        }

        protected override String PreprocessResult(String s_actual)
        {
            if (s_actual == null) return null;
            return s_actual.SplitLines().Select(ln =>
            {
                if (ln.Trim().IsEmpty()) return ln;
                var indent = Seq.Nats.First(i => ln[i] != ' ');
                var s_indent = ln.Slice(0, indent);
                ln = ln.Slice(indent);
                while (ln.Contains("  ")) ln = ln.Replace("  ", " ");
                return s_indent + ln;
            }).StringJoin(Environment.NewLine);
        }
    }

    [TestFixture]
    public abstract class BaseCubinTests : BaseTests
    {
        protected override String ReferenceNamespace()
        {
            return typeof(TestRunner).Namespace + ".Reference";
        }

        protected override ReadOnlyCollection<String> ReferenceWannabes()
        {
            return base.ReferenceWannabes().Select(wb => wb + ".cubin").ToReadOnly();
        }
    }
}