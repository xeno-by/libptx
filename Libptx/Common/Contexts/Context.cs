using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Functions;
using Libptx.Instructions;
using Libptx.Statements;
using XenoGears.Traits.Disposable;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Common.Contexts
{
    [DebuggerNonUserCode]
    public class Context
    {
        public Module Module { get; private set; }
        public SoftwareIsa Version { get { return Module == null ? SoftwareIsa.PTX_21 : Module.Version; } }
        public HardwareIsa Target { get { return Module == null ? HardwareIsa.SM_10 : Module.Target; } }
        public bool UnifiedTexturing { get { return Module == null ? true : Module.UnifiedTexturing; } }
        public bool DowngradeDoubles { get { return Module == null ? false : Module.DowngradeDoubles; } }

        public Stack<Atom> Stack { get; private set; }
        public Atom Parent { get { return Stack.SecondOrDefault(); } }
        public Entry Entry { get { return Stack.OfType<Entry>().SingleOrDefault2(); } }
        public Statement Stmt { get { return Stack.OfType<Statement>().SingleOrDefault2(); } }
        public ptxop Ptxop { get { return Stack.OfType<ptxop>().SingleOrDefault2(); } }
        public Label Label { get { return Stack.OfType<Label>().SingleOrDefault2(); } }

        public HashSet<Atom> Visited { get; private set; }
        public HashSet<Statement> VisitedStmts { get; private set; }
        public HashSet<Expression> VisitedExprs { get; private set; }

        public Context(Module module)
        {
            Module = module;

            Stack = new Stack<Atom>();
            Visited = new HashSet<Atom>();
            VisitedStmts = new HashSet<Statement>();
            VisitedExprs = new HashSet<Expression>();
        }

        protected virtual void CorePush(Atom atom)
        {
        }

        protected virtual void CorePop(Atom atom)
        {
            Visited.Add(atom);
            if (atom is Statement) VisitedStmts.Add((Statement)atom);
            if (atom is Expression) VisitedExprs.Add((Expression)atom);
        }

        public IDisposable Push(Atom atom)
        {
            Stack.Push(atom);
            CorePush(atom);

            return new DisposableAction(() =>
            {
                (Stack.Peek() == atom).AssertTrue();
                Stack.Pop();
                CorePop(atom);
            });
        }
    }
}