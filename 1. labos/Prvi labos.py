from abc import ABC, abstractmethod
from itertools import product
from numpy import prod


class DomainElement:

    def __init__(self, *values):
        self.values = list(values)

    def getNumberOfComponents(self) -> int:
        return len(self.values)

    def getComponentValue(self) -> int:
        return self.values

    def hashCode(self) -> int:
        pass

    def equals(self, other) -> bool:
        return other.getComponentValue() == self.getComponentValue()

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

    def iterator(self):
        pass

    def getCardinality(self):
        return prod([simpleDomain.getCardinality() for simpleDomain in self.SimpleDomains])

    def getComponent(self, index: int):
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
    def getDomain(self):
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
        return "\n" + "\n".join([f"d({i})={value:.6f}" for i, value in zip(range(self.IDomain.getFirst(),
                                                                                self.IDomain.getLast()),self.memberships)])

    def set(self, domainElement, double):
        if (index := self.IDomain.indexOfElement(domainElement)) != -1:
            self.memberships[index] = double

    def getDomain(self):
        return self.IDomain

    def getValueAt(self, domainElement) -> float:
        return self.memberships[domainElement.getComponentValue()[0]]


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


if __name__ == "__main__":
    d1 = Domain.intRange(0, 5)
    print(d1)
    print(f"Kardinalitet domene je: {d1.getCardinality()}")

    d2 = Domain.intRange(0, 3)
    print(d2)
    print(f"Kardinalitet domene je: {d2.getCardinality()}")

    d3 = Domain.combine(d1, d2)
    print(d3)
    print(f"Kardinalitet domene je: {d3.getCardinality()}")

    print(d3.elementForIndex(0))
    print(d3.elementForIndex(5))
    print(d3.elementForIndex(14))
    print(d3.indexOfElement(DomainElement.of(4,1)))

    d = Domain.intRange(0, 11)
    set1 = MutableFuzzySet(d)
    set1.set(DomainElement.of(0), 1.0)
    set1.set(DomainElement.of(1), 0.8)
    set1.set(DomainElement.of(2), 0.6)
    set1.set(DomainElement.of(3), 0.4)
    set1.set(DomainElement.of(4), 0.2)
    print("Set1", set1)

    d2_F = Domain.intRange(-5, 6)
    set2 = CalculatedFuzzySet(d2_F, StandardFuzzySets.lambdaFunction(
                d2_F.indexOfElement(DomainElement.of(-4)),
                d2_F.indexOfElement(DomainElement.of(0)),
                d2_F.indexOfElement(DomainElement.of(4))
                ))
    print("Set2", set2)

    notSet1 = Operations.unaryOperation(set1, Operations.zadehNot())
    print("NotSet1", notSet1)

    union = Operations.binaryOperation(set1, notSet1, Operations.zadehOr())
    print("Union", union)

    hinters = Operations.binaryOperation(set1, notSet1, Operations.hamacherTNorm(1.0))
    print("Set1 intersection with notSet1 using parameterised Hamacher T norm with parameter 1.0:", hinters)
