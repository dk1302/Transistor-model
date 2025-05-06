import random

params = [4.57, 4.95, 1.0, 2.0, 2.0, 3.0, 0.03, 0.1, 0.06, 0.02, 0.02, 0.03, 5.0, 5.0, 300.0, 3.9, 0.5]
temp = params.copy()
for x in range(1000):
    file = open(f"input/file{x-1}.inp", "w", newline='')
    file.write(f"""go atlas simflags=\"-P 1\"
TITLE   MMT poly-Si simulation for pixel
set WF={params[0]}
set WFG={params[1]}
set LDDvar={params[2]}
set Dvar={params[3]}
set Svar={params[4]}
set Lvar={params[5]}
set SCvar={params[6]}
set Toxvar={params[7]}
set Toxvar2={params[8]}
set Evar={params[9]}
set FPhvar={params[10]}
set OHvar={params[11]}
set CG2var={params[12]}
set CG1var={params[13]}
set Tempvar={params[14]}
set Poxvar={params[15]}
set Mode2dope={params[16]}
#Schottky contact for eg 1um into S
mesh width=1               
#
x.m     l=0                  spac=($"Svar"/5)
x.m     l=$"Svar"          spac=0.01
x.m     l=($"Svar"+0.5)        spac=0.1
x.m     l=($"Svar"+1)     spac=0.01
x.m     l=(($"Svar"+$"Lvar"+$"Dvar")/2)    spac=($"Lvar"/10)
x.m     l=($"Svar"+$"Lvar")      spac=($"Lvar"/10)
x.m     l=($"Svar"+$"Lvar"+$"LDDvar")  spac=($"LDDvar"/10)
x.m     l=($"Svar"+$"Lvar"+$"LDDvar"+$"Dvar")    spac=($"Dvar"/10)
#
#
#y.m        l=(0-$"Evar"-$"Toxvar2"-0.1)     spac=0.02
y.m     l=(0-$"Evar"-$"Toxvar2")         spac=($"Toxvar2"/5)
y.m     l=(0-$"Toxvar2")       spac=($"Toxvar2"/5)
#y.m        l=(0-($"Evar"*2))      spac=($"Evar"/2)
#y.m        l=(0-$"FPhvar"-$"Evar")      spac=0.1
y.m     l=(0-$"Evar")          spac=0.005
#y.m        l=(0-$"FPhvar")            spac=($"FPhvar"/2)
y.m     l=0              spac=($"SCvar"/10)
y.m     l=$"SCvar"             spac=($"SCvar"/10)
y.m     l=($"SCvar"+$"Toxvar")       spac=($"Toxvar"/2)
y.m     l=($"SCvar"+$"Toxvar"+$"Evar")     spac=$"Evar"
y.m     l=($"SCvar"+$"Toxvar"+$"Evar"+$"OHvar")  spac=$"Evar"
#y.m        l=($"SCvar"+$"Toxvar"+($"Evar"*2)+$"OHvar") spac=$"Evar"
#y.m        l=($"toxvar"+$"SCvar"+0.2)       spac=0.05
#
#
#  **************** regions ****************
#             1=oxide  2=silicon 3-oxide
#
region       num=1  y.min=(0-$"Evar"-$"Toxvar2"-0.1)  y.max=0    oxide
region       num=2  y.min=0   y.max=$"SCvar"  silicon
region       num=3  y.min=$"SCvar"  y.max=($"toxvar"+$"SCvar"+0.2)   oxide
#
#  **************** electrodes ****************
#
elec  num=1  x.min=0    x.max=($"Svar"+1)  \
         y.min=($"SCvar"+$"Toxvar") y.max=($"SCvar"+$"Toxvar"+$"Evar") name=CG1
#elec  num=1  x.min=($"Svar"-0.5)    x.max=($"Svar"+1)  \
         y.min=$"SCvar"+$"Toxvar"+$"Evar"+$"OHvar" y.max=$"SCvar"+$"Toxvar"+($"Evar"*2)+$"OHvar" name=CG1
#elec  num=1  x.min=$"Svar"-0.5    x.max=$"Svar"  \
         y.min=$"SCvar"+$"Toxvar"+$"Evar" y.max=$"SCvar"+$"Toxvar"+$"Evar"+$"OHvar" name=CG1
####put back#### +$"Evar") name=gate
elec  num=2  x.min=($"Svar"+1)   x.max= $"Svar"+$"Lvar"+$"LDDvar" \
         y.min=(0-$"Toxvar2"-$"Evar") y.max= (0-$"Toxvar2") name=CG2
#
elec  num=3  x.min=0    x.max=$"Svar"  y.min=0-$"Evar"   y.max=0  name=source
elec  num=3  x.min=$"Svar"-0.5 x.max=$"Svar"+0.5 y.min=0-$"Evar"*2 y.max=0-$"Evar" name=source
###not used elec  num=3  x.min=$"Svar"    x.max=($"Svar"+$"FPvar") \ 
#        y.min=(0-$"FPhvar"-$"Evar")   y.max=(0-$"FPhvar")  name=source
elec  num=4  x.min=$"Svar"+$"Lvar"+$"LDDvar"    x.max=$"Svar"+$"Lvar"+$"Lddvar"+$"Dvar" \
         y.min=0-$"Evar"   y.max=0  name=drain
#elec  num=4  x.min=$"Svar"+$"Lvar"-0.5 x.max=$"Svar"+$"Lvar"+0.5 y.min=0-$"Evar"*2 y.max=0-$"Evar" name=drain
### not used elec  num=4  x.min=(($"Svar"+$"Lvar")-$"FPvar")  x.max=($"Svar"+$"Lvar") \
#        y.min=(0-$"FPhvar"-$"Evar")  y.max=(0-$"FPhvar") name=drain
 
 
#  **************** doping profiles  ****************
#doping     reg=2  uniform conc=3.e20 n.type x.left=($"Svar"+$"Lvar")  char=0.3
#doping     reg=2  uniform conc=3.e20 n.type x.right=($"Svar"-$"Mode2dope")  char=0.3
#doping from mutsu
doping     reg=2  uniform conc=4.6e17 n.type x.left=($"Svar"+$"Lvar")  char=0.3
doping     reg=2  uniform conc=2.3e19 n.type x.left=($"Svar"+$"Lvar"+$"LDDvar")  char=0.3
material material=silicon mun=300 mup=30 eg300=1.12 affinity=4.17 taun0=1e-8 taup0=1e-8 nc300=2.5e20 nv300=2.5e20
#10 / 5
#defects nta=1.12e21 ntd=4.e20 wta=0.025 wtd=0.1 \
#nga=1.e18 ngd=3.e18 ega=0.4 egd=0.4 wga=0.1 wgd=0.1 \
#sigtae=1.e-16 sigtah=1.e-14 sigtde=1.e-14 sigtdh=1.e-16 \
#siggae=1.e-16 siggah=1.e-14 siggde=1.e-14 siggdh=1.e-16 continuous
#2003 PDF
defects nta=0.8e20 ntd=1.5e20 wta=0.05 wtd=0.03 \
nga=7.e19 ngd=1.e20 ega=0.08 egd=0.1 wga=0.1 wgd=0.1 \
sigtae=1.e-16 sigtah=1.e-14 sigtde=1.e-14 sigtdh=1.e-16 \
siggae=1.e-16 siggah=1.e-14 siggde=1.e-14 siggdh=1.e-16 continuous
interface qf=-2e10
material region=1 PERMIT=$"Poxvar"
material region=3 PERMIT=$"Poxvar"
# define Mo gate workfunction, high WF shifts transfers to right for better Vg=0
#contact    name=gate workfunction=$"WFG"
contact num=1 workfunction=$"WFG"
contact num=2 workfunction=$"WFG"
contact num=4 workfunction=4.17 SURF.REC BARRIER ALPHA=2.7e-7 BETA=0
contact num=3 workfunction=$"WF" SURF.REC BARRIER ALPHA=2.7e-7 BETA=0
#BETA=0 
#contact name=drain workfunction=$"WF" SURF.REC BARRIER ALPHA=2.7e-7 BETA=0 
######################################## for simulation workfunction=4.52
# define models, include bbt.kl for reverse leakage effect with default parameters
models temp=$"Tempvar" trap.tunnel trap.coulombic srh bbt.kl
#srh bb.a=5e14 bb.b=1.9e7 bb.gamma=2.5
#
impact selb
method newton trap maxtrap=30   
output CON.BAND E.VELOCITY EX.VELOCITY EY.VELOCITY FLOWLINES H.VELOCITY HX.VELOCITY HY.VELOCITY QFN  VAL.BAND
solve init 
method  newton 
output CON.BAND E.VELOCITY EX.VELOCITY EY.VELOCITY FLOWLINES H.VELOCITY HX.VELOCITY HY.VELOCITY QFN VAL.BAND
#outputs 10vcg2
solve init
solve vstep=0.5 vfinal=10 name=CG2
solve vstep=-0.5 vfinal=0 name=drain
solve vstep=0.25 vfinal=1 name=CG1 
log outf=output{x-1}v1.log
solve vdrain=0
solve vstep=0.1 vfinal=10 name=drain
save outf=drain{x-1}v1.str
log off
solve vstep=0.5 vfinal=10 name=CG2
solve vstep=-0.5 vfinal=0 name=drain
solve vstep=0.25 vfinal=5 name=CG1 
log outf=output{x-1}v5.log
solve vdrain=0
solve vstep=0.1 vfinal=10 name=drain
save outf=drain{x-1}v5.str
log off
solve vstep=0.5 vfinal=10 name=CG2
solve vstep=-0.5 vfinal=0 name=drain
solve vstep=0.25 vfinal=10 name=CG1 
log outf=output{x-1}v10.log
solve vdrain=0
solve vstep=0.1 vfinal=10 name=drain
save outf=drain{x-1}v10.str
quit""")
    file.close()
    for y in range(len(params)):
        if(random.random() < 0.5):
            params[y] = random.uniform(temp[y] - (temp[y] * 0.2), temp[y] + (temp[y] * 0.2))