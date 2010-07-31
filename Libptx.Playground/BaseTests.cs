using System;
using System.Collections.ObjectModel;
using System.Linq;
using NUnit.Framework;
using XenoGears.Functional;
using XenoGears.Strings;

namespace Libptx.Playground
{
    [TestFixture]
    public class BasePtxTests : XenoGears.Playground.Framework.BaseTests
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
    public class BaseCubinTests : XenoGears.Playground.Framework.BaseTests
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