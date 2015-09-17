-- Practical 5
-- University of OXFORD

-- Implement LSTM from Graves' paper
-- i_t = \sigma(linear(x_t, h_{t-1}))
-- f_t = \sigma(linear(x_t, h_{t-1}))
-- c_t = f_t * c_{t-1} + i_t * tanh(linear(x_t, h_{t-1})
-- o_t = \sigma(linear(x_t, h_{t-1}))
-- h_t = o_t * tanh(c_t)

torch.manualSeed(0)
require 'nngraph'
local c = require 'trepl.colorize'

-- Node values (and size)
n_x = 5; T = 10
xv = {}
for t = 1, T do
   xv[t] = torch.randn(n_x)
end

-- Graphical model definition
nngraph.setDebug(true)
local x_t  = nn.Identity()()
x_t:annotate{graphAttributes = {color = 'red', fontcolor = 'red'}}
local h_tt = nn.Identity()() -- h_tt := h_{t-1}
h_tt:annotate{graphAttributes = {color = 'red', fontcolor = 'red'}}
local c_tt = nn.Identity()() -- c_tt := c_{t-1}
c_tt:annotate{graphAttributes = {color = 'red', fontcolor = 'red'}}

n_h = 4
n_i, n_f, n_o, n_c = n_h, n_h, n_h, n_h

local i_t = nn.Sigmoid()(nn.CAddTable()({
   nn.Linear(n_x, n_i)(x_t),
   nn.Linear(n_h, n_i)(h_tt)
}))
i_t:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}

local f_t = nn.Sigmoid()(nn.CAddTable()({
   nn.Linear(n_x, n_f)(x_t),
   nn.Linear(n_h, n_f)(h_tt)
}))
f_t:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}

local cc_t = nn.Tanh()(nn.CAddTable()({
   nn.Linear(n_x, n_c)(x_t),
   nn.Linear(n_h, n_c)(h_tt)
}))
cc_t:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}

local c_t = nn.CAddTable()({
   nn.CMulTable()({f_t, c_tt}),
   nn.CMulTable()({i_t, cc_t})
})
c_t:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}}

local o_t = nn.Sigmoid()(nn.CAddTable()({
   nn.Linear(n_x, n_o)(x_t),
   nn.Linear(n_h, n_o)(h_tt),
   nn.Linear(n_c, n_o)(c_t)
}))
o_t:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}

local h_t = nn.CMulTable()({o_t, nn.Tanh()(c_t)} )
h_t:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}}

nngraph.annotateNodes()
LSTM_module = nn.gModule({c_tt, h_tt, x_t}, {c_t, h_t})

--pcall(function()
   inTable = {}
   outTable = {}
   outTable[0] = {torch.zeros(n_c), torch.zeros(n_h)}
   for i = 1, #xv do
      table.insert(inTable, {outTable[i-1][1], outTable[i-1][2], xv[i]})
      table.insert(outTable, LSTM_module:forward(inTable[i]))
   end
--end)
graph.dot(LSTM_module.fg, 'LSTM', 'LSTM')

-- Call as getParameters(20) if 20 is still a Linear
function getParameters(node)
   local model = LSTM_module
   for a, b in ipairs(model.forwardnodes) do
      if b.id == node then
         print(c.green('Node ' .. node .. ': ' .. tostring(b.data.module)))
         print(c.blue('\nWeights:'))
         print(b.data.module.weight)
         print(c.blue('Bias:'))
         print(b.data.module.bias)
         return
      end
   end
end

function showTimeT(t)
   if t > T then
      print(c.red('t > T = ' .. T))
   else
      print(c.green('Time t = ' .. t))
      print(c.magenta('Inputs'))
      print(c.blue('c['..tostring(t-1)..']:'))
      print(inTable[t][1])
      print(c.blue('h['..tostring(t-1)..']:'))
      print(inTable[t][2])
      print(c.blue('x['..t..']:'))
      print(inTable[t][3])

      print(c.magenta('Outputs'))
      print(c.blue('c['..t..']:'))
      print(outTable[t][1])
      print(c.blue('h['..t..']:'))
      print(outTable[t][2])
   end
end

function printfile(node,fname)
   local model = LSTM_module
   file = io.open(fname .. ".txt","w")
   file2 = io.open(fname .. "Bias.txt","w")
   for a, b in ipairs(model.forwardnodes) do
      if b.id == node then
        --file:write("Node" .. node .. ": " .. tostring(b.data.module) .."\n")
        --file:write("Weight\n")
        for _, data in ipairs(b.data.module.weight:storage():totable()) do
           file:write(tostring(math.floor(data * 256)) .. ',')
        end
        --file:write("\nBias\n")
        for _, data in ipairs(b.data.module.bias:totable()) do
           file2:write(tostring(math.floor(data * 256)) .. ',')
        end
        break
      end
   end
   file:close()
   file2:close()
end

function printinput(t)
   file = io.open("input.txt","w")
   for _, data in ipairs(inTable[t][3]:totable()) do
        file:write(tostring(math.floor(data * 256)) .. ',')
   end
   file:close()
   file1 = io.open("c_tt.txt","w")
   for _, data in ipairs(inTable[t][1]:totable()) do
        file1:write(tostring(math.floor(data * 256)) .. ',')
   end
   file1:close()
   file2 = io.open("h_tt.txt","w")
   for _, data in ipairs(inTable[t][2]:totable()) do
        file2:write(tostring(math.floor(data * 256)) .. ',')
   end
   file2:close()
   file3 = io.open("output.txt","w")
   file3:write("H_o\n" .. tostring(torch.floor(outTable[t][1]*256)))
   file3:write("C_o\n" .. tostring(torch.floor(outTable[t][2]*256)))
   file3:close()
end


print [[

If `20` is a `nn.Linear()` node, then print its weight with
   getParameters(20)

Print all inputs and outputs at time 0 with
   showTimeT(1)

printinput will write inputs and outputs at t in a text file
   printinput(1)
printfile will write a node parameters in a text file.
The bias of the node will be in a different file name.
   printfile(20,"Wf")
]]
