﻿using Libcuda.DataTypes;
using Libptx.Expressions.Slots.Specials.Annotations;

namespace Libptx.Expressions.Slots.Specials
{
    [Special("%tid", typeof(uint4))]
    public partial class tid : Special
    {
    }
}