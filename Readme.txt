Provides object model and infrastructure for reading, writing and manipulating codes of PTX assembly language.

Implemented functionality:
 * Data structures, enumerations and other boileplate necessary to provide strongly-typed object model.
 * ISA reference: a bunch of classes with annotated properties that reflect modifiers and operands of PTX instructions.
 * Validator that doesn't let syntactically and/or semantically invalid code pass undetected.
 * Renderer that emits PTX modules in text format.

Not implemented stuff:
 * Embedded DSL for emitting PTX (so that one can write stuff like "mul.lo.u32(foo, bar, qux)" and enjoy compile-time verification and intellisense).
 * Funcs (that's too much to take care about at the moment: complications of object model, various ABIs to support).
 * Indirect branches and calls (I don't have an SM_20 device, so can't test this functionality).
 * Scopes (this implies scoping outside entries, i.e. embedded blocks, local variables, extern/visible variables).
 * Debugging directives (I didn't have enough time to dig into DWARF format).
 * Bit bucket operands (For now that's too much work to find out what instructions support them and what instructions do not).

Quick facts:
 * Libcuda is built against PTX ISA 2.1 (but supports previous versions as well).
 * Unlike Libcuda this library hasn't passed rigorous testing (I've only scratched basic ALU, mem and flow instructions; misc, textures, video and synchronization instructions weren't touched at all).
 * Performance is awful because most logic is pretty reflection-heavy (this is gonna change soon due to caching that I'm going to implement).
 * Error propagation and reporting leaves much to be desired (in most cases one will need a stack trace to find out what's wrong).
 * Debugging the library needs to be performed with Just My Code turned off (I've overdone stuff with [DebuggerNonUserCode] and need to think what to do now).
