EXE := all_reduce_perf

default: $(EXE)

clean:
	rm -f $(EXE) *.o

%.o: %.cc 
	$(CXX) -o $@ $(CXXFLAGS) -c $<

$(EXE): main.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) $(LIBS)


