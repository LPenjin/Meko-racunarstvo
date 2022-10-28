from abc import ABC, abstractmethod
from itertools import product
from numpy import prod
import numpy as np


class DomainElement:

    def __init__(self, *values):
        self.values = list(values)

    def getNumberOfComponents(self) -> int:
        return len(self.values)

    def getComponentValue(self, int) -> int:
        return self.values[int]

    def hashCode(self) -> int:
        pass

    def equals(self, other) -> bool:
        return other.values == self.values

    def __str__(self):
        if len(self.values) == 1:
            return "(" + str(self.values[0]) + ")"
        else:
            return "(" + ",".join([str(element) for element in self.values]) + ")"

    @staticmethod
    def of(*args: int):
        return DomainElement(*args)


class IDomain(ABC):

    @abstractmethod
    def getCardinality(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def getComponent(self, index: int):
        raise NotImplementedError

    @abstractmethod
    def getNumberOfComponents(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def indexOfElement(self, DomainElement) -> int:
        raise NotImplementedError

    @abstractmethod
    def elementForIndex(self, n: int):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class Domain(IDomain):

    def __init__(self):
        pass

    @staticmethod
    def intRange(min: int, max: int):
        return SimpleDomain(min, max)

    @staticmethod
    def combine(domain1, domain2):
        return CompositeDomain(domain1, domain2)

    def indexOfElement(self, DomainElement):
        pass

    def elementForIndex(self, index: int):
        pass


class SimpleDomain(Domain):

    def __init__(self, first: int, last: int):
        super().__init__()
        self.first = first
        self.last = last

    def __str__(self):
        return "\n".join(["Element domene: " + str(i) for i in range(self.first, self.last)])

    def __iter__(self):
        self.cur = self.first
        return self

    def __next__(self):
        if self.cur < self.last:
            x = DomainElement.of(self.cur)
            self.cur += 1
            return x
        else:
            raise StopIteration

    def getCardinality(self) -> int:
        return self.last - self.first

    def getComponent(self, index: int):
        return self

    def getNumberOfComponents(self) -> int:
        return 1

    def iterator(self):
        pass

    def getFirst(self) -> int:
        return self.first

    def getLast(self) -> int:
        return self.last

    def elementForIndex(self, index: int):
        return DomainElement(self.first + index)

    def indexOfElement(self, domainElement):
        if domainElement.getNumberOfComponents() > 1:
            return -1
        else:
            elements = [DomainElement(i) for i in range(self.first, self.last)]
            for i, element in enumerate(elements):
                if element.equals(domainElement):
                    return i
            return -1


class CompositeDomain(Domain):

    def __init__(self, *args):
        super().__init__()
        self.SimpleDomains = list(args)

    def __str__(self):
        elements = [[i for i in range(simpleDomain.getFirst(), simpleDomain.getLast())] for simpleDomain in self.SimpleDomains]
        cartesianProducts = list(product(*elements))
        return "\n".join(["Element domene: " + str(cartesianProduct) for cartesianProduct in cartesianProducts])

    def __iter__(self):
        self.cur = 0
        return self

    def __next__(self):
        if self.cur < self.getCardinality():
            x = DomainElement.of(*self.elementForIndex(self.cur).values)
            self.cur += 1
            return x
        else:
            raise StopIteration

    def iterator(self):
        pass

    def getCardinality(self):
        return prod([simpleDomain.getCardinality() for simpleDomain in self.SimpleDomains])

    def getComponent(self, index: int) -> IDomain:
        return self.SimpleDomains[index]

    def getNumberOfComponents(self) -> int:
        return len(self.SimpleDomains)

    def elementForIndex(self, index: int):
        elements = [[i for i in range(simpleDomain.getFirst(), simpleDomain.getLast())] for simpleDomain in
                    self.SimpleDomains]
        return DomainElement(*list(product(*elements))[index])

    def indexOfElement(self, domainElementTest):
        if domainElementTest.getNumberOfComponents() != self.getNumberOfComponents():
            return 0
        else:
            elements = list(product(*[[i for i in range(simpleDomain.getFirst(), simpleDomain.getLast())] for simpleDomain in
                        self.SimpleDomains]))
            domainElements = [DomainElement(*element) for element in elements]
            for i, domainElement in enumerate(domainElements):
                if domainElement.equals(domainElementTest):
                    return i
            return 0


class IFuzzySet(ABC):

    @abstractmethod
    def getDomain(self) -> IDomain:
        pass

    @abstractmethod
    def getValueAt(self, domainElement) -> float:
        pass


class IInitUnaryFunction(ABC):

    def __init__(self, funct):
        self.funct = funct

    def valueAt(self, e: int) -> float:
        return self.funct(e)


class CalculatedFuzzySet(IFuzzySet):

    def __init__(self, IDomain, IInitUnaryFunction):
        self.IDomain = IDomain
        self.IIinitUnaryFunction = IInitUnaryFunction

    def __str__(self):
        return "\n" + "\n".join([f"d({value})={self.valueAt(value):.6f}" for value in range(self.IDomain.getFirst(),
                                                                                self.IDomain.getLast())])
    def getDomain(self):
        return self.IDomain

    def getValueAt(self, DomainElement):
        pass

    def valueAt(self, e: int) -> float:
        return self.IIinitUnaryFunction.valueAt(self.IDomain.indexOfElement(DomainElement.of(e)))


class MutableFuzzySet(IFuzzySet):

    memberships = []

    def __init__(self, IDomain):
        self.IDomain = IDomain
        self.memberships = [0.0 for i in range(IDomain.getCardinality())]

    def __str__(self):
        if self.IDomain.getNumberOfComponents() == 1:
            return "\n" + "\n".join([f"d({i})={value:.6f}" for i, value in zip(range(self.IDomain.getFirst(),
                                                                                self.IDomain.getLast()),self.memberships)])
        else:
            elements = [[i for i in range(self.IDomain.getComponent(simpleDomain_index).getFirst(),
                                          self.IDomain.getComponent(simpleDomain_index).getLast())]
                        for simpleDomain_index in range(self.IDomain.getNumberOfComponents())]
            cartesianProducts = list(product(*elements))
            return "\n".join([f"d{str(cartesianProduct)} = {membership:.6f}" for membership, cartesianProduct in zip(self.memberships, cartesianProducts)])

    def set(self, domainElement, double):
        if (index := self.IDomain.indexOfElement(domainElement)) != -1:
            self.memberships[index] = double

    def getDomain(self):
        return self.IDomain

    def getValueAt(self, domainElement) -> float:
        return self.memberships[self.getDomain().indexOfElement(domainElement)]


class StandardFuzzySets:

    def __init__(self):
        pass

    @staticmethod
    def lFuntion(a, b):
        def funct(x):
            if x < a:
                x = 0
            elif x >= b:
                x = 0
            else:
                x = (b-x) / (b-a)
        return IInitUnaryFunction(funct)

    @staticmethod
    def gammaFunction(a, b):
        def funct(x):
            if x < a:
                x = 0
            elif x >= b:
                x = 0
            else:
                x = (x-a)/(b-a)
            return x
        return IInitUnaryFunction(funct)

    @staticmethod
    def lambdaFunction(a, b, c):
        def funct(x):
            if x < a:
                x = 0.000000
            elif x > c:
                x = 0.000000
            elif x <= b and x >= a:
                x = (x-a)/(b-a)
            else:
                x = (c-x)/(c-a)
            return x
        return IInitUnaryFunction(funct)


class IBinaryFunction(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def valueAt(self, a, b):
        pass


class IUnaryFunction(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def valueAt(self, a):
        pass


class Operations(IUnaryFunction, IBinaryFunction):

    def __init__(self):
        pass

    @staticmethod
    def unaryOperation(set1, function):
        new_set = MutableFuzzySet(Domain.intRange(set1.getDomain().getFirst(), set1.getDomain().getLast()))
        for i in range(set1.getDomain().getFirst(), set1.getDomain().getLast()):
            new_set.set(DomainElement.of(i), function.valueAt(set1.getValueAt(DomainElement.of(i))))
        return new_set

    @staticmethod
    def binaryOperation(IFuzzySet1, IFuzzySet2, IBinaryFunction):
        first = min(IFuzzySet1.getDomain().getFirst(), IFuzzySet2.getDomain().getFirst())
        last = max(IFuzzySet1.getDomain().getLast(), IFuzzySet2.getDomain().getLast())
        new_set = MutableFuzzySet(Domain.intRange(first, last))
        for i in range(first, last):
            new_set.set(DomainElement.of(i), IBinaryFunction.valueAt(IFuzzySet1.getValueAt(DomainElement.of(i)),
                                                                     IFuzzySet2.getValueAt(DomainElement.of(i))))
        return new_set

    class zadehNot(IUnaryFunction):

        @staticmethod
        def valueAt(a):
            return 1 - a

    class zadehAnd(IBinaryFunction):

        @staticmethod
        def valueAt(a, b):
            return min(a, b)

    class zadehOr(IBinaryFunction):

        @staticmethod
        def valueAt(a, b):
            return max(a, b)

    class hamacherTNorm(IBinaryFunction):

        def __init__(self, v):
            self.v = v

        def valueAt(self, a, b):
            return (a*b)/(self.v + (1-self.v)*(a+b-a*b))

    class hamacherSNorm(IBinaryFunction):

        def __init__(self, v):
            self.v = v

        def valueAt(self, a, b):
            return (a+b-(2-self.v)*a*b)/(1-(1-self.v)*a*b)


class Relations:

    @staticmethod
    def isSymmetric(set1) -> bool:
        if Relations.isUtimesURelation(set1):
            arr = np.array(set1.memberships)
            d = set1.getDomain().getComponent(0).getCardinality()
            arr = arr.reshape(d, d)
            if (arr==arr.T).all():
                return True
        return False


    @staticmethod
    def isReflexive(set1) -> bool:
        if Relations.isUtimesURelation(set1):
            arr = np.array(set1.memberships)
            d = set1.getDomain().getComponent(0).getCardinality()
            arr = arr.reshape(d, d)
            for i in range(d):
                if arr[i, i] != 1:
                    return False
            return True
        return False

    @staticmethod
    def isMaxMinTransitive(set1) -> bool:
        if Relations.isUtimesURelation(set1):
            for i in range(set1.getDomain().getCardinality()):
                element = set1.getDomain().elementForIndex(i)
                set_range = range(set1.getDomain().getComponent(1).getFirst(), set1.getDomain().getComponent(1).getLast())
                mins = [min(set1.getValueAt(DomainElement.of(element.getComponentValue(0), j)),
                            set1.getValueAt(DomainElement.of(j, element.getComponentValue(1)))) for j in set_range]
                if set1.getValueAt(element) < max(mins):
                    return False
            return True
        return False

    @staticmethod
    def compositionOfBinaryRelations(set1, set2):
        new_set = MutableFuzzySet(Domain.combine(set1.getDomain().getComponent(0), set2.getDomain().getComponent(1)))
        element1 = range(set1.getDomain().getComponent(0).getFirst(), set1.getDomain().getComponent(0).getLast())
        element2 = range(set2.getDomain().getComponent(1).getFirst(), set2.getDomain().getComponent(1).getLast())
        element3 = range(set2.getDomain().getComponent(0).getFirst(), set2.getDomain().getComponent(0).getLast())
        for x in element1:
            for z in element2:
                new_set.set(DomainElement.of(x, z),
                            max([min(set1.getValueAt(DomainElement.of(x, y)),
                                     set2.getValueAt(DomainElement.of(y, z)))
                                 for y in element3]))
        return new_set

    @staticmethod
    def isFuzzyEquivalence(set1):
        return Relations.isUtimesURelation(set1) and Relations.isReflexive(set1) and Relations.isSymmetric(set1)\
                                                 and Relations.isMaxMinTransitive(set1)

    @staticmethod
    def isUtimesURelation(set1):
        if set1.getDomain().getNumberOfComponents() == 2:
            domain1 = set1.getDomain().getComponent(0)
            domain2 = set1.getDomain().getComponent(1)
            if domain1.getCardinality() == domain2.getCardinality() and domain1.getFirst() == domain2.getFirst():
                return True
        return False



if __name__ == "__main__":
    #1. zadatak
    #a)
    d1 = Domain.intRange(0, 5)
    print(d1)
    print(f"Kardinalitet domene je: {d1.getCardinality()}")
    print()

    d2 = Domain.intRange(0, 3)
    print(d2)
    print(f"Kardinalitet domene je: {d2.getCardinality()}")
    print()

    d3 = Domain.combine(d1, d2)
    print(d3)
    print(f"Kardinalitet domene je: {d3.getCardinality()}")
    print()

    print(d3.elementForIndex(0))
    print(d3.elementForIndex(5))
    print(d3.elementForIndex(14))
    print(d3.indexOfElement(DomainElement.of(4,1)))
    print()

    #b)
    d = Domain.intRange(0, 11)
    set1 = MutableFuzzySet(d)
    set1.set(DomainElement.of(0), 1.0)
    set1.set(DomainElement.of(1), 0.8)
    set1.set(DomainElement.of(2), 0.6)
    set1.set(DomainElement.of(3), 0.4)
    set1.set(DomainElement.of(4), 0.2)
    print("Set1", set1)
    print()

    d2_F = Domain.intRange(-5, 6)
    set2 = CalculatedFuzzySet(d2_F, StandardFuzzySets.lambdaFunction(
                d2_F.indexOfElement(DomainElement.of(-4)),
                d2_F.indexOfElement(DomainElement.of(0)),
                d2_F.indexOfElement(DomainElement.of(4))
                ))
    print("Set2", set2)
    print()

    #c)
    notSet1 = Operations.unaryOperation(set1, Operations.zadehNot())
    print("NotSet1", notSet1)
    print()

    union = Operations.binaryOperation(set1, notSet1, Operations.zadehOr())
    print("Union", union)
    print()

    hinters = Operations.binaryOperation(set1, notSet1, Operations.hamacherTNorm(1.0))
    print("Set1 intersection with notSet1 using parameterised Hamacher T norm with parameter 1.0:", hinters)
    print()

    #2. zadatak
    #a)
    u = Domain.intRange(1, 6)

    u2 = Domain.combine(u, u)

    r1 = MutableFuzzySet(u2)
    r1.set(DomainElement.of(1,1), 1)
    r1.set(DomainElement.of(2,2), 1)
    r1.set(DomainElement.of(3,3), 1)
    r1.set(DomainElement.of(4,4), 1)
    r1.set(DomainElement.of(5,5), 1)
    r1.set(DomainElement.of(3,1), 0.5)
    r1.set(DomainElement.of(1,3), 0.5)

    r2 = MutableFuzzySet(u2)
    r2.set(DomainElement.of(1,1), 1)
    r2.set(DomainElement.of(2,2), 1)
    r2.set(DomainElement.of(3,3), 1)
    r2.set(DomainElement.of(4,4), 1)
    r2.set(DomainElement.of(5,5), 1)
    r2.set(DomainElement.of(3,1), 0.5)
    r2.set(DomainElement.of(1,3), 0.1)

    r3 = MutableFuzzySet(u2)
    r3.set(DomainElement.of(1, 1), 1)
    r3.set(DomainElement.of(2, 2), 1)
    r3.set(DomainElement.of(3, 3), 0.3)
    r3.set(DomainElement.of(4, 4), 1)
    r3.set(DomainElement.of(5, 5), 1)
    r3.set(DomainElement.of(1, 2), 0.6)
    r3.set(DomainElement.of(2, 1), 0.6)
    r3.set(DomainElement.of(2, 3), 0.7)
    r3.set(DomainElement.of(3, 2), 0.7)
    r3.set(DomainElement.of(3, 1), 0.5)
    r3.set(DomainElement.of(1, 3), 0.5)

    r4 = MutableFuzzySet(u2)
    r4.set(DomainElement.of(1, 1), 1)
    r4.set(DomainElement.of(2, 2), 1)
    r4.set(DomainElement.of(3, 3), 1)
    r4.set(DomainElement.of(4, 4), 1)
    r4.set(DomainElement.of(5, 5), 1)
    r4.set(DomainElement.of(1, 2), 0.4)
    r4.set(DomainElement.of(2, 1), 0.4)
    r4.set(DomainElement.of(2, 3), 0.5)
    r4.set(DomainElement.of(3, 2), 0.5)
    r4.set(DomainElement.of(1, 3), 0.4)
    r4.set(DomainElement.of(3, 1), 0.4)

    test1 = Relations.isUtimesURelation(r1)
    print("r1 je definiran nad UxU?", test1)

    test2 = Relations.isSymmetric(r1)
    print("r1 je simetrican", test2)

    test3 = Relations.isSymmetric(r2)
    print("r2 je simetrican", test3)

    test4 = Relations.isReflexive(r1)
    print("r1 je refleksivan", test4)

    test5 = Relations.isReflexive(r3)
    print("r3 je refleksivan", test5)

    test6 = Relations.isMaxMinTransitive(r3)
    print("r3 je tranzitivan", test6)

    test7 = Relations.isMaxMinTransitive(r4)
    print("r4 je tranzitivan", test7)

    print()

    #b)
    u1 = Domain.intRange(1, 5)
    u2 = Domain.intRange(1, 4)
    u3 = Domain.intRange(1, 5)

    r1 = MutableFuzzySet(Domain.combine(u1, u2))
    r1.set(DomainElement.of(1, 1), 0.3)
    r1.set(DomainElement.of(1, 2), 1)
    r1.set(DomainElement.of(3, 3), 0.5)
    r1.set(DomainElement.of(4, 3), 0.5)

    r2 = MutableFuzzySet(Domain.combine(u2, u3))
    r2.set(DomainElement.of(1, 1), 1)
    r2.set(DomainElement.of(2, 1), 0.5)
    r2.set(DomainElement.of(2, 2), 0.7)
    r2.set(DomainElement.of(3, 3), 1)
    r2.set(DomainElement.of(3, 4), 0.4)

    r1r2 = Relations.compositionOfBinaryRelations(r1, r2)
    #print("r1r2", r1r2)

    for e in r1r2.getDomain():
        print(f"m({e})={r1r2.getValueAt(e)}")
    print()

    #c)
    u = Domain.intRange(1, 5)

    r = MutableFuzzySet(Domain.combine(u, u))
    r.set(DomainElement.of(1, 1), 1)
    r.set(DomainElement.of(2, 2), 1)
    r.set(DomainElement.of(3, 3), 1)
    r.set(DomainElement.of(4, 4), 1)
    r.set(DomainElement.of(1, 2), 0.3)
    r.set(DomainElement.of(2, 1), 0.3)
    r.set(DomainElement.of(2, 3), 0.5)
    r.set(DomainElement.of(3, 2), 0.5)
    r.set(DomainElement.of(3, 4), 0.2)
    r.set(DomainElement.of(4, 3), 0.2)

    r2 = r

    print("Početna relacija je neizrazita relacija ekvivalencije? ", Relations.isFuzzyEquivalence(r2))

    for i in range(3):
        r2 = Relations.compositionOfBinaryRelations(r2, r)

        print(f"Broj odrađenih kompozicija: {i}. Relacija je:")

        for e in r2.getDomain():
            print(f"m({e})={r2.getValueAt(e)}")

        print("Ova relacija je neizrazita relacija ekvivalencije? ", Relations.isFuzzyEquivalence(r2))
        print()


