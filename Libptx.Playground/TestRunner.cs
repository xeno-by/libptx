using System;
using Libptx.Reflection;
using System.Linq;
using XenoGears.Functional;

namespace Libptx.Playground
{
    internal class TestRunner
    {
        public static void Main(String[] args)
        {
            Ptxops.All.Select(t => t.PtxopSigs()).Ping();

//            // see more details at http://www.nunit.org/index.php?p=consoleCommandLine&r=2.5.5
//            var nunitArgs = new List<String>();
//            nunitArgs.Add("/run:Libptx.Playground");
//            nunitArgs.Add("/include:Hot");
//            nunitArgs.Add("/domain:None");
//            nunitArgs.Add("/noshadow");
//            nunitArgs.Add("/nologo");
//            nunitArgs.Add("Libptx.Playground.exe");
//            NUnit.ConsoleRunner.Runner.Main(nunitArgs.ToArray());
        }
    }
}
