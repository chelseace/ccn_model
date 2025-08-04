# Units 
module my_units
    using Unitful
    export μM, ms, s, hr, day
    @unit μM " μM" μM (10^(-6))*u"mol" true
    @unit ms " ms" ms (10^(-3))*u"s" true
    
	const μM = u"μM"
	const ms = u"ms"
    const s = u"s"
    const hr = u"hr"
    const day = u"d"  

    Unitful.register(@__MODULE__)
    function __init__()
        return Unitful.register(@__MODULE__)
    end               
end