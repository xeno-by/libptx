using System;
using System.Collections.ObjectModel;
using System.Linq;
using NUnit.Framework;
using XenoGears.Functional;

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