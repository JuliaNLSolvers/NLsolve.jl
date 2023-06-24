trace(r::SolverResults) = r.trace
function x_trace(r::SolverResults)
    tr = trace(r).states
    !haskey(tr[1].metadata, "x") && error("Trace does not contain x. To get a trace of x, run nlsolve() with extended_trace = true")
    [ state.metadata["x"] for state in tr ]
end
function F_trace(r::SolverResults)
    tr = trace(r).states
    !haskey(tr[1].metadata, "f(x)") && error("Trace does not contain F. To get a trace of the residuals, run nlsolve() with extended_trace = true")
    [ state.metadata["f(x)"] for state in tr ]
end
function J_trace(r::SolverResults)
    tr = trace(r).states
    !haskey(tr[1].metadata, "g(x)") && error("Trace does not contain J. To get a trace of the Jacobian, run nlsolve() with extended_trace = true")
    [ state.metadata["g(x)"] for state in tr ]
end