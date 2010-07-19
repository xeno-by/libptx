using Libptx.Expressions;

namespace Libptx.Edsl.Vars
{
    public class var_couple : VarCouple
    {
        public var_pred fst { get; set; }
        public var_pred snd { get; set; }
    }
}